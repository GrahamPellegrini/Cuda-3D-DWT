#include <fstream>
#include <vector>
#include <string>
#include <iostream>

// Function to write a 3D vector to the file
void write_3d_vector(std::ofstream& file, const std::vector<std::vector<std::vector<float>>>& vec, int d1, int d2, int r1, int r2, int c1, int c2)
{   
    // Loop through the 3D vector and write each element to the file in depth, rows, cols order
    for (; d1 < d2; ++d1) {
        for (; r1 < r2; ++r1) {
            for (; c1 < c2; ++c1) {
                // Write the element to the file
                file << vec[d1][r1][c1] << " ";
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
            depth /= 2;
            rows /= 2;
            cols /= 2;

            // Write the level
            file << "Level " << level << std::endl;
            // Write the dimensions
            file << depth << " " << rows << " " << cols << std::endl;


            // Write the LLL bands to the file
            file << "LLL" << std::endl;
            write_3d_vector(file, output, 0, depth, 0, rows, 0, cols);
            // Write the LLH bands to the file
            file << "LLH" << std::endl;
            write_3d_vector(file, output, 0, depth, 0, rows, cols, 2 * cols);
            // Write the LHL bands to the file
            file << "LHL" << std::endl;
            write_3d_vector(file, output, 0, depth, rows, 2 * rows, 0, cols);
            // Write the LHH bands to the file
            file << "LHH" << std::endl;
            write_3d_vector(file, output, 0, depth, rows, 2 * rows, cols, 2 * cols);
            // Write the HLL bands to the file
            file << "HLL" << std::endl;
            write_3d_vector(file, output, depth, 2 * depth, 0, rows, 0, cols);
            // Write the HLH bands to the file
            file << "HLH" << std::endl;
            write_3d_vector(file, output, depth, 2 * depth, 0, rows, cols, 2 * cols);
            // Write the HHL bands to the file
            file << "HHL" << std::endl;
            write_3d_vector(file, output, depth, 2 * depth, rows, 2 * rows, 0, cols);
            // Write the HHH bands to the file
            file << "HHH" << std::endl;
            write_3d_vector(file, output, depth, 2 * depth, rows, 2 * rows, cols, 2 * cols);
        }
        break;
    default:
        std::cerr << "Invalid level: " << levels << std::endl;
        break;
    }
}