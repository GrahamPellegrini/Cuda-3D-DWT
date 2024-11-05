#ifndef LOADBIN_H
#define LOADBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

std::vector<std::vector<std::vector<float>>> loadvolume(const std::string& filename) {
    int depth, rows, cols;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read the dimensions
    file.read(reinterpret_cast<char*>(&depth), sizeof(depth));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Initialize the 3D vector
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    // Read the data
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            file.read(reinterpret_cast<char*>(volume[d][r].data()), cols * sizeof(float));
        }
    }

    // Check if the volume is empty
    if (volume.empty()) {
        throw std::runtime_error("Volume is empty.");
    }
    else {
        std::cerr << "Volume loaded successfully from " << filename << std::endl;
        std::cerr << "Depth: " << depth << ", Rows: " << rows << ", Cols: " << cols << std::endl;
    }

    file.close();
    return volume;
}

#endif // LOADBIN_H