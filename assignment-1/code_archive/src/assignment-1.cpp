#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <chrono>
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

// Function to perform 1D DWT on a signal
void dwt_1d(std::vector<float>& signal, int db_num) {
    // Retrieve the low-pass and high-pass filters based on db_num
    const auto& low_filter = db_low[db_num - 1];
    const auto& high_filter = db_high[db_num - 1];
    int filter_length = low_filter.size();

    // Prepare vector for DWT output
    std::vector<float> transformed(signal.size(), 0.0f);

    // Perform convolution and downsampling
    for (std::vector<float>::size_type i = 0; i <= signal.size() - filter_length; i += 2) {
        float low_sum = 0.0f, high_sum = 0.0f;

        // Convolve signal with filters
        for (int j = 0; j < filter_length; ++j) {
            low_sum += signal[i + j] * low_filter[j];
            high_sum += signal[i + j] * high_filter[j];
        }

        // Store results in transformed vector
        transformed[i / 2] = low_sum;
        transformed[i / 2 + signal.size() / 2] = high_sum;
    }

    // Copy transformed result back into signal
    signal = transformed;
}

// Function to perform 3D DWT on a 3D volume
void dwt_3D(std::vector<std::vector<std::vector<float>>>& volume, int db_num) {
    // Get the shape of the volume
    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    // Iterate over each depth level
    for (int d = 0; d < depth; ++d) {
        // Apply 1D DWT to each row
        for (int i = 0; i < rows; ++i) {
            dwt_1d(volume[d][i], db_num);
        }

        // Apply 1D DWT to each column
        for (int j = 0; j < cols; ++j) {
            std::vector<float> column(rows);
            for (int i = 0; i < rows; ++i) {
                column[i] = volume[d][i][j];
            }
            dwt_1d(column, db_num);
            for (int i = 0; i < rows; ++i) {
                volume[d][i][j] = column[i];
            }
        }
    }

    // Iterate over each row and column
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::vector<float> signal(depth);
            for (int d = 0; d < depth; ++d) {
                signal[d] = volume[d][i][j];
            }
            // Apply 1D DWT to the depth
            dwt_1d(signal, db_num);
            for (int d = 0; d < depth; ++d) {
                volume[d][i][j] = signal[d];
            }
        }
    }

    
}

// Function to perform multi-level 3D DWT on a 3D volume
void multi_level (std::vector<std::vector<std::vector<float>>>& volume, int db_num, int levels) {
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

            // Rezise the volume to the new dimensions which captures the approximation coefficients (LLL)
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
    // std::cerr << "Program completed successfully" << std::endl;
    
    // Log the time taken for the program
    std::cerr << "Total time taken: " << t.count()<< " seconds" << std::endl;

    return EXIT_SUCCESS;
}