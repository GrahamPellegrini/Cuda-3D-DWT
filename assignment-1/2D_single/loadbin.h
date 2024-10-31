#ifndef LOADBIN_H
#define LOADBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

std::vector<std::vector<float>> loadslice(const std::string& filename) {
    int rows, cols;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read the dimensions
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Debug prints to verify dimensions
    std::cerr << "Read dimensions: " << rows << "x" << cols << std::endl;

    // Initialize the 2D vector
    std::vector<std::vector<float>> slice(rows, std::vector<float>(cols));

    // Read the data
    for (int r = 0; r < rows; ++r) {
        file.read(reinterpret_cast<char*>(slice[r].data()), cols * sizeof(float));
        // Debug print to verify data read for each row
        std::cerr << "Read row " << r << ": ";
        for (float val : slice[r]) {
            std::cerr << val << " ";
        }
        std::cerr << std::endl;
    }

    file.close();
    return slice;
}

#endif // LOADBIN_H