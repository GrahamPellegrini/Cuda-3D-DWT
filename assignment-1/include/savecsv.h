#ifndef SAVECSV_H
#define SAVECSV_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Function to write a 3D vector to the file
void write_3d_vector(std::ofstream& file, const std::vector<std::vector<std::vector<float>>>& vec) 
{   
    // Loop through the 3D vector and write each element to the file in depth, rows, cols order
    for (int d = 0; d < vec.size(); ++d) {
        for (int r = 0; r < vec[d].size(); ++r) {
            for (int c = 0; c < vec[d][r].size(); ++c) {
                // Write the element to the file
                file << vec[d][r][c] << " ";
            }
            // Write a new line after each row
            file << std::endl;
        }
    }
}
// Take the computed coefficients and save them to a CSV file to be used in visualization and analysis in python
void save_coefficients_to_csv(const std::string& csv_file,
                              const std::vector<std::vector<std::vector<float>>>& LLL,
                              const std::vector<std::vector<std::vector<float>>>& LLH,
                              const std::vector<std::vector<std::vector<float>>>& LHL,
                              const std::vector<std::vector<std::vector<float>>>& LHH,
                              const std::vector<std::vector<std::vector<float>>>& HLL,
                              const std::vector<std::vector<std::vector<float>>>& HLH,
                              const std::vector<std::vector<std::vector<float>>>& HHL,
                              const std::vector<std::vector<std::vector<float>>>& HHH)
{
    // Open the file for writing
    std::ofstream file(csv_file);
    
    // Check if the file is being opened
    if (!file.is_open()) {
        // Error message if the file is not being opened (debugging)
        std::cerr << "Error opening file: " << csv_file << std::endl;
        return;
    }
    
    // Write the shape of the volume, asssuming all sub-bands have the same shape. Which should be the case
    int rows = LLL.size();
    int cols = LLL[0].size();
    int depth = LLL[0][0].size();
    // Write the shape of the volume to the first line of the file
    file << rows << " " << cols << " " << depth << std::endl;
    
    // Write each sub-band to the file
    file << "LLL" << std::endl;
    write_3d_vector(file, LLL);
    file << "LLH" << std::endl;
    write_3d_vector(file, LLH);
    file << "LHL" << std::endl;
    write_3d_vector(file, LHL);
    file << "LHH" << std::endl;
    write_3d_vector(file, LHH);
    file << "HLL" << std::endl;
    write_3d_vector(file, HLL);
    file << "HLH" << std::endl;
    write_3d_vector(file, HLH);
    file << "HHL" << std::endl;
    write_3d_vector(file, HHL);
    file << "HHH" << std::endl;
    write_3d_vector(file, HHH);
    // Close the file after writing
    file.close();
}

#endif // SAVECSV_H