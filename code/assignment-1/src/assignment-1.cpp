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
    int n = signal.size();
    L.resize(n / 2);
    H.resize(n / 2);
    
    for (int i = 0; i < n / 2; ++i) {
        L[i] = (signal[2 * i] + signal[2 * i + 1]) / std::sqrt(2.0);
        H[i] = (signal[2 * i] - signal[2 * i + 1]) / std::sqrt(2.0);
    }
}

// 3D Haar wavelet transform
void haar_wavelet_3d(std::vector<std::vector<std::vector<float>>>& volume,
                     std::vector<std::vector<std::vector<float>>>& LLL,
                     std::vector<std::vector<std::vector<float>>>& LLH,
                     std::vector<std::vector<std::vector<float>>>& LHL,
                     std::vector<std::vector<std::vector<float>>>& LHH,
                     std::vector<std::vector<std::vector<float>>>& HLL,
                     std::vector<std::vector<std::vector<float>>>& HLH,
                     std::vector<std::vector<std::vector<float>>>& HHL,
                     std::vector<std::vector<std::vector<float>>>& HHH) {
    int rows = volume.size();
    int cols = volume[0].size();
    int depth = volume[0][0].size();
    
    std::cerr << "Volume dimensions: " << rows << " x " << cols << " x " << depth << std::endl;

    // Temporary storage for intermediate results
    std::vector<std::vector<std::vector<float>>> temp(rows, std::vector<std::vector<float>>(cols, std::vector<float>(depth)));
    
    // Apply 1D Haar transform along z dimension (depth)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::vector<float> L, H;
            haar_wavelet_1d(volume[i][j], L, H);
            for (int k = 0; k < depth / 2; ++k) {
                temp[i][j][k] = L[k];
                temp[i][j][k + depth / 2] = H[k];
            }
        }
    }
    
    // Apply 1D Haar transform along y dimension (cols)
    std::vector<std::vector<std::vector<float>>> temp2(rows, std::vector<std::vector<float>>(cols, std::vector<float>(depth)));
    for (int k = 0; k < depth; ++k) {
        for (int i = 0; i < rows; ++i) {
            std::vector<float> L, H;
            std::vector<float> signal(cols);
            for (int j = 0; j < cols; ++j) {
                signal[j] = temp[i][j][k];
            }
            haar_wavelet_1d(signal, L, H);
            for (int j = 0; j < cols / 2; ++j) {
                temp2[i][j][k] = L[j];
                temp2[i][j + cols / 2][k] = H[j];
            }
        }
    }
    
    // Apply 1D Haar transform along x dimension (rows)
    LLL.resize(rows / 2, std::vector<std::vector<float>>(cols / 2, std::vector<float>(depth / 2)));
    LLH.resize(rows / 2, std::vector<std::vector<float>>(cols / 2, std::vector<float>(depth / 2)));
    LHL.resize(rows / 2, std::vector<std::vector<float>>(cols / 2, std::vector<float>(depth / 2)));
    LHH.resize(rows / 2, std::vector<std::vector<float>>(cols / 2, std::vector<float>(depth / 2)));
    HLL.resize(rows / 2, std::vector<std::vector<float>>(cols / 2, std::vector<float>(depth / 2)));
    HLH.resize(rows / 2, std::vector<std::vector<float>>(cols / 2, std::vector<float>(depth / 2)));
    HHL.resize(rows / 2, std::vector<std::vector<float>>(cols / 2, std::vector<float>(depth / 2)));
    HHH.resize(rows / 2, std::vector<std::vector<float>>(cols / 2, std::vector<float>(depth / 2)));

    for (int k = 0; k < depth / 2; ++k) {
        for (int i = 0; i < rows / 2; ++i) {
            for (int j = 0; j < cols / 2; ++j) {
                LLL[i][j][k] = temp2[i][j][k];
                LLH[i][j][k] = temp2[i][j + cols / 2][k];
                LHL[i][j][k] = temp2[i][j][k + depth / 2];
                LHH[i][j][k] = temp2[i][j + cols / 2][k + depth / 2];
                HLL[i][j][k] = temp2[i + rows / 2][j][k];
                HLH[i][j][k] = temp2[i + rows / 2][j + cols / 2][k];
                HHL[i][j][k] = temp2[i + rows / 2][j][k + depth / 2];
                HHH[i][j][k] = temp2[i + rows / 2][j + cols / 2][k + depth / 2];
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

    std::string csv_file = argv[1];
    std::string output_coefficients = argv[2];
    int rows, cols, depth;
    // Print hello
    std::cerr << "Hello" << std::endl;
    auto volume = load_volume_from_csv(csv_file, rows, cols, depth);

    std::cerr << "Loaded volume dimensions: " << rows << " x " << cols << " x " << depth << std::endl;

    std::vector<std::vector<std::vector<float>>> LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH;
    haar_wavelet_3d(volume, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH);
    
    // Save the transformed coefficients to a CSV file
    save_coefficients_to_csv(output_coefficients, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH);

    return EXIT_SUCCESS;
}
