#include <fstream>
#include <vector>
#include <string>
#include <iostream>

// Function to write a 3D vector to the file
void write_3d_vector(std::ofstream& file, const std::vector<std::vector<std::vector<float>>>& vec, int depth, int rows, int cols)
{   
    // Loop through the 3D vector and write each element to the file in depth, rows, cols order
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                // Write the element to the file
                file << vec[d][r][c] << " ";
            }
            // Write a new line after each row
            file << std::endl;
        }
    }
}

void save_csv(
const std::string& csv_file,
const std::vector<std::vector<std::vector<float>>>& output,
int depth, int rows, int cols, int levels
)
{
    // Open the file for writing
    std::ofstream file(csv_file);
    
    // Check if the file is being opened
    if (!file.is_open()) {
        // Error message if the file is not being opened (debugging)
        std::cerr << "Error opening file: " << csv_file << std::endl;
        return;
    }

    switch (levels)
    {
    case 1:
    case 2:
    case 3:
    case 4:
        for (int level = 1; level <= levels; ++level) {
            // Split the dimensions respectfully
            int d = depth >> level;
            int r = rows >> level;
            int c = cols >> level;

            // Write the level
            file << "Level " << level << std::endl;

            // Write the LLL bands to the file
            file << "LLL" << std::endl;
            write_3d_vector(file, output, d, r, c);
            // Write the LLH bands to the file
            file << "LLH" << std::endl;
            write_3d_vector(file, output, d, r, 2 * c);
            // Write the LHL bands to the file
            file << "LHL" << std::endl;
            write_3d_vector(file, output, d, 2 * r, c);
            // Write the LHH bands to the file
            file << "LHH" << std::endl;
            write_3d_vector(file, output, d, 2 * r, 2 * c);
            // Write the HLL bands to the file
            file << "HLL" << std::endl;
            write_3d_vector(file, output, 2 * d, r, c);
            // Write the HLH bands to the file
            file << "HLH" << std::endl;
            write_3d_vector(file, output, 2 * d, r, 2 * c);
            // Write the HHL bands to the file
            file << "HHL" << std::endl;
            write_3d_vector(file, output, 2 * d, 2 * r, c);
            // Write the HHH bands to the file
            file << "HHH" << std::endl;
            write_3d_vector(file, output, 2 * d, 2 * r, 2 * c);
        }
        break;
    default:
        std::cerr << "Invalid level: " << levels << std::endl;
        break;
    }
}