#ifndef LOADBIN_H
#define LOADBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cassert>

// Function to load a 3D volume from a binary file
std::vector<std::vector<std::vector<float>>> loadvolume(const std::string& filename) {
    // Initialize the variables
    int depth, rows, cols;
    // Open the file
    std::ifstream file(filename, std::ios::binary);

    // Check if the file is open
    assert(file.is_open() && "File not found.");

    // Read the dimensions from the binary file to the variables
    file.read(reinterpret_cast<char*>(&depth), sizeof(depth));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Initialize the 3D vector with the dimensions variables
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    // Read the data iteratively from the binary file to the 3D vector
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            file.read(reinterpret_cast<char*>(volume[d][r].data()), cols * sizeof(float));
        }
    }

    // Check if the volume is empty
    assert(!volume.empty() && "Volume is empty."); 
    assert(depth > 0 && rows > 0 && cols > 0 && "Invalid dimensions."); 


    file.close();
    return volume;
}

#endif // LOADBIN_H