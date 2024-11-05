#ifndef SAVEBIN_H
#define SAVEBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cassert>

// Save a 3D volume to a binary file
void savevolume(const std::vector<std::vector<std::vector<float>>>& volume, const std::string& filename) {
    // Open the file
    std::ofstream file(filename, std::ios::binary);

    // Check if the file is open
    assert(file.is_open() && "Error opening file");

    // Get the dimensions of the volume
    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    // Write the dimensions to the binary file
    file.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Itteratively write the volume to the binary file
    for (const auto& slice : volume) {
        for (const auto& row : slice) {
            file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
        }
    }

    // Show the file dimensions written and check if the file is written correctly
    assert(file.tellp() == sizeof(depth) + sizeof(rows) + sizeof(cols) + depth * rows * cols * sizeof(float) && "Error writing to file");
    
    file.close();
}

#endif // SAVEBIN_H