#ifndef SAVEBIN_H
#define SAVEBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

void saveslice(const std::vector<std::vector<float>>& slice, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int rows = slice.size();
    int cols = slice[0].size();

    // Write the dimensions
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Write the data
    for (const auto& row : slice) {
        file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
    }

    file.close();
}

#endif // SAVEBIN_H