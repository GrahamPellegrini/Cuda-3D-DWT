#ifndef SAVEBIN_H
#define SAVEBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

void savevolume(const std::vector<std::vector<std::vector<float>>>& volume, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    // Write the dimensions
    file.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Write the data
    for (const auto& slice : volume) {
        for (const auto& row : slice) {
            file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
        }
    }

    // Check if the write was successful
    if (!file.good()) {
        throw std::runtime_error("Error writing to file: " + filename);
    }
    else {
        std::cerr << "Volume saved to " << filename << std::endl;
        std::cerr << "Dimensions: " << depth << "x" << rows << "x" << cols << std::endl;
    }

    file.close();
}

#endif // SAVEBIN_H