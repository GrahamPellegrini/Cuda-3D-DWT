#ifndef LOADCSV_H
#define LOADCSV_H

#include <iostream> 
#include <fstream> 
#include <sstream>  
#include <string> 
#include <vector>
#include <algorithm>

// Load a 3D volume from the python formated CSV file
std::vector<std::vector<std::vector<float>>> load_volume_from_csv(const std::string& csv_file, int& rows, int& cols, int& depth)
{
    // Opening the file and initializing the line 
    std::ifstream file(csv_file);
    std::string line;
    
    // Check if the file is being opened and line is being read
    if (!std::getline(file, line)) {
        // Error message if the file is not being read (debugging)
        std::cerr << "Error reading the first line of the CSV file." << std::endl;
        return {};
    }
    // Showing the first line of csv, which is formated for dimensions (debugging)
    std::cerr << "First line: " << line << std::endl; 
    
    // Replace commas with spaces to handle comma-separated values
    std::replace(line.begin(), line.end(), ',', ' ');
    
    //Take the dimensions of the volume from the first line
    std::istringstream shape_stream(line);
    // Check if the dimensions are being read
    if (!(shape_stream >> depth >> rows >> cols)) {
        // Error message if the dimensions are not being read (debugging)
        std::cerr << "Error parsing the shape of the volume from the first line." << std::endl;
        return {};
    }
    
    // Verify the first line read of dimensions (debugging)
    std::cerr << "Read shape: " << depth << " x " << rows << " x " << cols << std::endl;
    
    // Check if the dimensions are valid
    if (rows <= 0 || cols <= 0 || depth <= 0) {
        // Error message if the dimensions are invalid (debugging)
        std::cerr << "Invalid volume dimensions: " << rows << " x " << cols << " x " << depth << std::endl;
        return {};
    }
    
    // Initialize the 3D vector with the given dimensions
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));
    // Note representation of 3D shape with vectors is gonna take lots of computation to iterate through, thats why later implementation will use cuda to parrallelize the computation
    
    // Iterate through the depth, rows, and columns to read the values into the volume
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            // Check if the line is being read
            if (!std::getline(file, line)) {
                // Error message if the line is not being read (debugging)
                std::cerr << "Error reading line for depth " << d << ", row " << r << std::endl;
                return {};
            }
            // Line by line replacement of commas with spaces
            std::replace(line.begin(), line.end(), ',', ' ');
            // Initialize the row stream to read the values 
            std::istringstream row_stream(line);
            // Iterate through the columns to read the values into the volume
            for (int c = 0; c < cols; ++c) {
                // Check if the values are being read
                if (!(row_stream >> volume[d][r][c])) {
                    // Error message if the values are not being read (debugging)
                    std::cerr << "Error parsing value at depth " << d << ", row " << r << ", col " << c << std::endl;
                    return {};
                }
            }
            // Check if there are extra values in the row
            if (row_stream >> line) {
                // Error message if there are extra values in the row (debugging)
                std::cerr << "Extra values found in depth " << d << ", row " << r << std::endl;
                return {};
            }
        }
    }
    // After all the iterations, return the volume
    return volume;
}
// Note the extensive checks ensure that the load function handles all possible errors that may occur during the reading of the csv file. However, these if statments are being currently used for debugging purposes and till the csv file format and the reading align. These ifs will be removed in the final implementation, as they take lots of extra computation time.

#endif // LOADCSV_H