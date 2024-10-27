#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "../include/loadcsv.h"
#include "../include/savecsv.h"

// 1D Haar wavelet transform
void haar_wavelet_1d(const std::vector<float>& signal, std::vector<float>& L, std::vector<float>& H) {
    // Get the size of the signal
    int n = signal.size();
    // Set the size of the output vectors for the low-pass and high-pass coefficients
    L.resize(n / 2);
    H.resize(n / 2);
    
    // Loop through the signal and apply the Haar wavelet transform
    for (int i = 0; i < n / 2; ++i) {
        // Compute the low-pass and high-pass coefficients
        // The Haar wavelet transform is a simple difference and average operation for the low-pass and high-pass coefficients, respectively
        L[i] = (signal[2 * i] + signal[2 * i + 1]) / std::sqrt(2.0);// Average
        H[i] = (signal[2 * i] - signal[2 * i + 1]) / std::sqrt(2.0);// Difference
    }
}

// 3D Haar wavelet transform
void haar_wavelet_3d(std::vector<std::vector<std::vector<float>>>& volume,
                     int depth, int rows, int cols,
                     std::vector<std::vector<std::vector<float>>>& LLL,
                     std::vector<std::vector<std::vector<float>>>& LLH,
                     std::vector<std::vector<std::vector<float>>>& LHL,
                     std::vector<std::vector<std::vector<float>>>& LHH,
                     std::vector<std::vector<std::vector<float>>>& HLL,
                     std::vector<std::vector<std::vector<float>>>& HLH,
                     std::vector<std::vector<std::vector<float>>>& HHL,
                     std::vector<std::vector<std::vector<float>>>& HHH) {
  
    
    // Confirm the dimensions of the volume (debugging)
    std::cerr << "Volume dimensions: " << volume.size() << " x " << volume[0].size() << " x " << volume[0][0].size() << std::endl;

    // First temporary storage for intermediate results along the y dimension
    std::vector<std::vector<std::vector<float>>> temp(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    
    // Apply 1D Haar transform along y dimension (cols)
    // Iterate through the other dimensions
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            std::vector<float> L, H;
            // Apply the 1D Haar wavelet transform to the 1D signal along the y dimension
            haar_wavelet_1d(volume[d][r], L, H);
            // Iterate through the half of the columns to store the low-pass and high-pass coefficients
            for (int c = 0; c < cols / 2; ++c) {
                temp[d][r][c] = L[c];
                temp[d][r][c + cols / 2] = H[c];
            }
        }
    }
    
    // Second temporary storage for intermediate results along the x dimension
    std::vector<std::vector<std::vector<float>>> temp2(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    // Apply 1D Haar transform along x dimension (rows)
    // Iterate through the other dimensions
    for (int d = 0; d < depth; ++d) {
        for (int c = 0; c < cols; ++c) {
            std::vector<float> L, H;
            std::vector<float> signal(rows);
            // Store the already transformed coefficients from the temporary storage
            for (int r = 0; r < rows; ++r) {
                // Taking the 1D signal along the x dimension
                signal[r] = temp[d][r][c];
            }
            // Apply the 1D Haar wavelet transform to the 1D signal along the x dimension
            haar_wavelet_1d(signal, L, H);
            // Iterate through the half of the rows to store the low-pass and high-pass coefficients
            for (int r = 0; r < rows / 2; ++r) {
                temp2[d][r][c] = L[r];
                temp2[d][r + rows / 2][c] = H[r];
            }
        }
    }
    
    // iterate through the volume to store the final sub-bands by splitting the temp2 with the sub-bands logic
    for (int d = 0; d < depth / 2; ++d) {
        for (int r = 0; r < rows / 2; ++r) {
            for (int c = 0; c < cols / 2; ++c) {
                LLL[d][r][c] = temp2[d][r][c];
                LLH[d][r][c] = temp2[d][r][c + cols / 2];
                LHL[d][r][c] = temp2[d][r + rows / 2][c];
                LHH[d][r][c] = temp2[d][r + rows / 2][c + cols / 2];
                HLL[d][r][c] = temp2[d + depth / 2][r][c];
                HLH[d][r][c] = temp2[d + depth / 2][r][c + cols / 2];
                HHL[d][r][c] = temp2[d + depth / 2][r + rows / 2][c];
                HHH[d][r][c] = temp2[d + depth / 2][r + rows / 2][c + cols / 2];
            }
        }
    }
}

// Main program entry point
int main(int argc, char *argv[])
{
    std::cerr << "Assignment 1: Synchronous DWT on 3D CT Image" << std::endl;

    if (argc != 3) { // Expecting two arguments: directory and depth
        std::cerr << "Usage: " << argv[0] << " <csv_file> <output_coefficients>" << std::endl;
        return EXIT_FAILURE;
    }

    // Load the arguments into variables
    std::string csv_file = argv[1];
    std::string output_coefficients = argv[2];
    int rows, cols, depth;
    
    // Load the volume from the CSV file using the defined function in loadcsv.h
    auto volume = load_volume_from_csv(csv_file, rows, cols, depth);

    // Check if the volume is loaded (debugging)
    std::cerr << "Loaded volume dimensions: " <<depth << " x " << rows << " x " << cols << std::endl;

    // Define the 3D vectors to store the transformed coefficients with the respective sub-bands size
    std::vector<std::vector<std::vector<float>>> LLL(depth / 2, std::vector<std::vector<float>>(rows / 2, std::vector<float>(cols / 2)));
    std::vector<std::vector<std::vector<float>>> LLH(depth / 2, std::vector<std::vector<float>>(rows / 2, std::vector<float>(cols / 2)));
    std::vector<std::vector<std::vector<float>>> LHL(depth / 2, std::vector<std::vector<float>>(rows / 2, std::vector<float>(cols / 2)));
    std::vector<std::vector<std::vector<float>>> LHH(depth / 2, std::vector<std::vector<float>>(rows / 2, std::vector<float>(cols / 2)));
    std::vector<std::vector<std::vector<float>>> HLL(depth / 2, std::vector<std::vector<float>>(rows / 2, std::vector<float>(cols / 2)));
    std::vector<std::vector<std::vector<float>>> HLH(depth / 2, std::vector<std::vector<float>>(rows / 2, std::vector<float>(cols / 2)));
    std::vector<std::vector<std::vector<float>>> HHL(depth / 2, std::vector<std::vector<float>>(rows / 2, std::vector<float>(cols / 2)));
    std::vector<std::vector<std::vector<float>>> HHH(depth / 2, std::vector<std::vector<float>>(rows / 2, std::vector<float>(cols / 2)));

    // Apply the 3D Haar wavelet transform using the defined function
    haar_wavelet_3d(volume, depth, rows, cols, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH);
    
    // Save the transformed coefficients to a CSV file using the defined function in savecsv.h
    save_coefficients_to_csv(output_coefficients, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH);

    // Free the memory allocated for the volume and coefficients
    volume.clear();
    LLL.clear();
    LLH.clear();
    LHL.clear();
    LHH.clear();
    HLL.clear();
    HLH.clear();
    HHL.clear();
    HHH.clear();

    // Return success
    return EXIT_SUCCESS;
}
