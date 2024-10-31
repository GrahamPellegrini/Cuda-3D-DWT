#ifndef LOADCSV_H
#define LOADCSV_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept> // For std::runtime_error

// Load a 3D volume from a CSV file formatted in Python style
std::vector<std::vector<std::vector<float>>> load_csv(const std::string& csv_file, int& depth , int& rows, int& cols)
{
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + csv_file);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Error reading the first line of the CSV file.");
    }
    
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream shape_stream(line);
    if (!(shape_stream >> depth >> rows >> cols)) {
        throw std::runtime_error("Error parsing the shape of the volume from the first line.");
    }
    
    if (rows <= 0 || cols <= 0 || depth <= 0) {
        throw std::runtime_error("Invalid volume dimensions: " + std::to_string(rows) + " x " + std::to_string(cols) + " x " + std::to_string(depth));
    }
    
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            if (!std::getline(file, line)) {
                throw std::runtime_error("Error reading line for depth " + std::to_string(d) + ", row " + std::to_string(r));
            }
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row_stream(line);
            for (int c = 0; c < cols; ++c) {
                if (!(row_stream >> volume[d][r][c])) {
                    throw std::runtime_error("Error parsing value at depth " + std::to_string(d) + ", row " + std::to_string(r) + ", col " + std::to_string(c));
                }
            }
        }
    }

    return volume;
}

#endif // LOADCSV_H
