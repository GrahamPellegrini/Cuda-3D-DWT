#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <memory>
#include "../include/loadcsv.h"
#include "../include/savecsv.h"
#include "../../code/shared/jbutil.h"

// Hard coding wavelet low and high pass filters coefficients for different db wavelet types
const std::vector<float> db1L = {0.7071067811865476, 0.7071067811865476};
const std::vector<float> db1H = {-0.7071067811865476, 0.7071067811865476};
const std::vector<float> db2L = {-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416};
const std::vector<float> db2H = {-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037};
const std::vector<float> db3L = {0.03522629188570953, -0.08544127388202666, -0.13501102001025458, 0.45987750211849154, 0.8068915093110925, 0.33267055295008263};
const std::vector<float> db3H = {-0.33267055295008263, 0.8068915093110925, -0.45987750211849154, -0.13501102001025458, 0.08544127388202666, 0.03522629188570953};
const std::vector<float> db4L = {-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309, -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965};
const std::vector<float> db4H = {-0.2303778133088965, 0.7148465705529157, -0.6308807679298589, -0.027983769416859854, 0.18703481171909309, 0.030841381835560764, -0.0328830116668852, -0.010597401785069032};

// Function to get wavelet coefficients based on db_num
void get_wavelet_coefficients(int db_num, std::vector<float>& low_pass, std::vector<float>& high_pass) {
    switch (db_num) {
        case 1:
            low_pass = db1L;
            high_pass = db1H;
            break;
        case 2:
            low_pass = db2L;
            high_pass = db2H;
            break;
        case 3:
            low_pass = db3L;
            high_pass = db3H;
            break;
        case 4:
            low_pass = db4L;
            high_pass = db4H;
            break;
        default:
            throw std::invalid_argument("Invalid db_num, must be between 1 and 4");
    }
}

// 1D wavelet transform
void wavelet_1d(std::vector<float>& signal, int db_num = 1) {
    int n = signal.size();
    std::vector<float> temp(n);
    std::vector<float> low_pass, high_pass;
    int idx;

    // Get the appropriate wavelet coefficients
    get_wavelet_coefficients(db_num, low_pass, high_pass);

    // Iterate through half the signal length
    for (int i = 0; i < n / 2; ++i) {
        // Initialize the temporary signal for low and high pass values
        temp[i] = 0;
        temp[i + n / 2] = 0;
        // Note the high and low pass filters are symmetric (have the same size)
        for (int j = 0; j < low_pass.size(); ++j) {
            idx = (2 * i + j) % n;
            temp[i] += low_pass[j] * signal[idx];
            temp[i + n / 2] += high_pass[j] * signal[idx];
        }
    }

    // Copy the result back to the signal
    signal = temp; // Direct assignment
}

// 3D wavelet transform
void wavelet_3d(std::vector<std::vector<std::vector<float>>>& volume, int depth, int rows, int cols, int db_num = 1) {
    // Apply the 1D wavelet transform along the depth
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::vector<float> signal(depth);
            for (int d = 0; d < depth; ++d) {
                signal[d] = volume[d][r][c];
            }
            wavelet_1d(signal, db_num);
            for (int d = 0; d < depth; ++d) {
                volume[d][r][c] = signal[d];
            }
        }
    }

    // Apply the 1D wavelet transform along the rows
    for (int d = 0; d < depth; ++d) {
        for (int c = 0; c < cols; ++c) {
            std::vector<float> signal(rows);
            for (int r = 0; r < rows; ++r) {
                signal[r] = volume[d][r][c];
            }
            wavelet_1d(signal, db_num);
            for (int r = 0; r < rows; ++r) {
                volume[d][r][c] = signal[r];
            }
        }
    }

    // Apply the 1D wavelet transform along the columns
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            std::vector<float> signal(cols);
            for (int c = 0; c < cols; ++c) {
                signal[c] = volume[d][r][c];
            }
            wavelet_1d(signal, db_num);
            for (int c = 0; c < cols; ++c) {
                volume[d][r][c] = signal[c];
            }
        }
    }
}

// Multi-level wavelet transform
void multi_level_wavelet(
    std::vector<std::vector<std::vector<float>>>& volume,
    int& depth, int& rows, int& cols, int multi_level, int db_num) 
{

    // Start timer to measure time taken
    double t = jbutil::gettime();

    for (int i = 0; i < multi_level; ++i) 
    {
        // Perform 3D wavelet transform in-place
        wavelet_3d(volume, depth, rows, cols, db_num);

        if (i < multi_level - 1) {
            // Halve the dimensions for the next level without extra copying
            depth /= 2;
            rows /= 2;
            cols /= 2;
        }
    }


    // Stop timer
    t = jbutil::gettime() - t;
    // Show time taken
    std::cerr << "Time taken: " << t << "s" << std::endl;

}

// Main program entry point
int main(int argc, char *argv[]) {
    std::cerr << "Assignment 1: Synchronous DWT on 3D CT Image" << std::endl;

    if (argc != 5) { // Expecting four arguments: csv_file, output_coefficients, multi_level, db_num
        std::cerr << "Usage: " << argv[0] << " <csv_file> <output_coefficients> <multi_level> <db_num>" << std::endl;
        return EXIT_FAILURE;
    }

    // Load the arguments into variables
    std::string csv_file = argv[1];
    std::string output_coefficients = argv[2];
    int multi_level = std::stoi(argv[3]);
    int db_num = std::stoi(argv[4]);

    // Check if the multi_level is between 1 and 4
    if (multi_level < 1 || multi_level > 4) {
        std::cerr << "Error: multi_level must be between 1 and 4." << std::endl;
        return EXIT_FAILURE;
    }
    // Check if the db_num is between 1 and 4
    if (db_num < 1 || db_num > 4) {
        std::cerr << "Error: db_num must be between 1 and 4." << std::endl;
        return EXIT_FAILURE;
    }

    // Load the CT volume from CSV file
    std::vector<std::vector<std::vector<float>>> volume;
    int depth, rows, cols;
    volume = load_csv(csv_file, depth, rows, cols);

    // Check if the volume was loaded successfully (debugging)
    if (volume.empty()) {
        std::cerr << "Error loading the volume from CSV file." << std::endl;
        return EXIT_FAILURE;
    }
    else {
        std::cerr << "Volume loaded successfully." << std::endl;
        // Print the dimensions of the volume (debugging)
        std::cerr << "Volume dimensions: " << depth << " x " << rows << " x " << cols << std::endl;
    }

    // Apply the multi-level wavelet transform
    multi_level_wavelet(volume, depth, rows, cols, multi_level, db_num);

    // Save the wavelet coefficients to a CSV file
    save_csv(output_coefficients, volume, depth, rows, cols, multi_level);
    
    return EXIT_SUCCESS;
}
