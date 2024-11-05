#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "../include/loadbin.h"
#include "../include/savebin.h"
#include "../../code/shared/jbutil.h"

// Define the wavelet coefficients as floats
const std::vector<std::vector<float>> db_low = {
    {0.70710678f, 0.70710678f}, // db1
    {-0.12940952f, 0.22414387f, 0.83651630f, 0.48296291f}, // db2
    {0.03522629f, -0.08544127f, -0.13501102f, 0.45987750f, 0.80689151f, 0.33267055f}, // db3
    {-0.01059740f, 0.03288301f, 0.03084138f, -0.18703481f, -0.02798377f, 0.63088077f, 0.71484657f, 0.23037781f} // db4
};

const std::vector<std::vector<float>> db_high = {
    {-0.70710678f, 0.70710678f}, // db1
    {-0.48296291f, 0.83651630f, -0.22414387f, -0.12940952f}, // db2
    {-0.33267055f, 0.80689151f, -0.45987750f, -0.13501102f, 0.08544127f, 0.03522629f}, // db3
    {-0.23037781f, 0.71484657f, -0.63088077f, -0.02798377f, 0.18703481f, 0.03084138f, -0.03288301f, -0.01059740f} // db4
};

void dwt_1d(std::vector<float>& signal, int db_num) {
    if (db_num < 1 || db_num > 4) {
        std::cerr << "Invalid db_num. Please select from db1, db2, db3, or db4." << std::endl;
        return;
    }

    // Retrieve the low-pass and high-pass filters based on db_num
    const std::vector<float>& low_filter = db_low[db_num - 1];
    const std::vector<float>& high_filter = db_high[db_num - 1];
    int filter_length = low_filter.size();

    // Prepare vectors to store results temporarily
    int approx_size = (signal.size() + 1) / 2;
    std::vector<float> approx(approx_size, 0.0f);
    std::vector<float> detail(approx_size, 0.0f);

    // Perform convolution and downsampling
    int signal_length = signal.size();
    for (int i = 0; i <= signal_length - filter_length; i += 2) {
        float low_sum = 0.0f;
        float high_sum = 0.0f;

        // Apply filters
        for (int j = 0; j < filter_length; ++j) {
            low_sum += signal[i + j] * low_filter[j];
            high_sum += signal[i + j] * high_filter[j];
        }

        // Store the results in the approximation and detail vectors
        approx[i / 2] = low_sum;
        detail[i / 2] = high_sum;
    }

    // Copy the approximation and detail coefficients back to the original signal vector
    for (int i = 0; i < approx_size; ++i) {
        signal[i] = approx[i];
        signal[i + approx_size] = detail[i];
    }

    // Zero padding for remaining positions if signal length is odd
    for (int i = 2 * approx_size; i < signal_length; ++i) {
        signal[i] = 0.0f;
    }
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
// Main program entry point
int main(int argc, char *argv[]) {
    std::cerr << "Assignment 1: Synchronous DWT on 3D CT Image" << std::endl;

    if (argc != 4) { // Expecting 3 arguments: bin_in, bin_out, db_num
        std::cerr << "Usage: " << argv[0] << " <bin_in> <bin_out> <db_num>" << std::endl;
        return EXIT_FAILURE;
    }

    // Load the arguments into variables
    std::string bin_in = argv[1];
    std::string bin_out = argv[2];
    int db_num = std::stoi(argv[3]);

    // Check if the db_num is between 1 and 4
    if (db_num < 1 || db_num > 4) {
        std::cerr << "Error: db_num must be between 1 and 4." << std::endl;
        return EXIT_FAILURE;
    }

    // Load the 3D slice from the binary file
    int depth, rows, cols;
    std::vector<std::vector<std::vector<float>>> volume = loadvolume(bin_in);

    // Check if the volume is empty
    if (volume.empty()) {
        std::cerr << "Error: Empty volume." << std::endl;
        return EXIT_FAILURE;
    }
    else {
        depth = volume.size();
        rows = volume[0].size();
        cols = volume[0][0].size();
        std::cerr << "Read dimensions: " << depth << "x" << rows << "x" << cols << std::endl;
    }

    // Perform 3D DWT on the volume
    dwt_3D(volume, db_num);


    // Save the 3D volume to the binary file
    savevolume(volume, bin_out);

    return EXIT_SUCCESS;
}