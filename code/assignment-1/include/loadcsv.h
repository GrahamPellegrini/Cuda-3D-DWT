#ifndef LOADCSV_H
#define LOADCSV_H

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

std::vector<std::vector<std::vector<float>>> load_volume_from_csv(const std::string& csv_file, int& rows, int& cols, int& depth) {
    std::ifstream file(csv_file);
    std::string line;
    
    // Read the shape of the volume
    if (!std::getline(file, line)) {
        std::cerr << "Error reading the first line of the CSV file." << std::endl;
        return {};
    }
    
    std::cerr << "First line: " << line << std::endl; // Debugging output to show the first line
    
    // Replace commas with spaces to handle comma-separated values
    std::replace(line.begin(), line.end(), ',', ' ');
    
    std::istringstream shape_stream(line);
    if (!(shape_stream >> rows >> cols >> depth)) {
        std::cerr << "Error parsing the shape of the volume from the first line." << std::endl;
        return {};
    }
    
    // Debugging output to verify the shape
    std::cerr << "Read shape: " << rows << " x " << cols << " x " << depth << std::endl;
    
    if (rows <= 0 || cols <= 0 || depth <= 0) {
        std::cerr << "Invalid volume dimensions: " << rows << " x " << cols << " x " << depth << std::endl;
        return {};
    }
    
    // Initialize the 3D vector
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));
    
    // Read the remaining lines and populate the 3D vector
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!std::getline(file, line)) {
                std::cerr << "Error reading line for row " << r << ", col " << c << std::endl;
                return {};
            }
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row_stream(line);
            for (int d = 0; d < depth; ++d) {
                if (!(row_stream >> volume[d][r][c])) {
                    std::cerr << "Error parsing value at row " << r << ", col " << c << ", depth " << d << std::endl;
                    return {};
                }
            }
            // Check if there are extra values in the row
            if (row_stream >> line) {
                std::cerr << "Extra values found in row " << r << ", col " << c << std::endl;
                return {};
            }
        }
    }
    
    return volume;
}

#endif // LOADCSV_H