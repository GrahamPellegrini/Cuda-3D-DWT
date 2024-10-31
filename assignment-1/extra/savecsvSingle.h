#include <fstream>
#include <vector>
#include <string>
#include <iostream>

// Function to write a 3D vector to the file
void write_3d_vector(std::ofstream& file, const std::vector<std::vector<std::vector<float>>>& vec, int d1, int d2, int r1, int r2, int c1, int c2)
{   
    // Loop through the 3D vector and write each element to the file in depth, rows, cols order
    for (int d = d1; d < d2; ++d) {
        for (int r = r1; r < r2; ++r) {
            for (int c = c1; c < c2; ++c) {
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
int depth, int rows, int cols
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

        // Split the dimensions respectfully
        int current_depth = depth / 2;
        int current_rows = rows / 2;
        int current_cols = cols / 2;

        // Write the dimensions
        file << current_depth << " " << current_rows << " " << current_cols << std::endl;

        // Write the LLL bands to the file
        file << "LLL" << std::endl;
        write_3d_vector(file, output, 0, current_depth, 0, current_rows, 0, current_cols);
        // Write the LLH bands to the file
        file << "LLH" << std::endl;
        write_3d_vector(file, output, 0, current_depth, 0, current_rows, current_cols, 2 * current_cols);
        // Write the LHL bands to the file
        file << "LHL" << std::endl;
        write_3d_vector(file, output, 0, current_depth, current_rows, 2 * current_rows, 0, current_cols);
        // Write the LHH bands to the file
        file << "LHH" << std::endl;
        write_3d_vector(file, output, 0, current_depth, current_rows, 2 * current_rows, current_cols, 2 * current_cols);
        // Write the HLL bands to the file
        file << "HLL" << std::endl;
        write_3d_vector(file, output, current_depth, 2 * current_depth, 0, current_rows, 0, current_cols);
        // Write the HLH bands to the file
        file << "HLH" << std::endl;
        write_3d_vector(file, output, current_depth, 2 * current_depth, 0, current_rows, current_cols, 2 * current_cols);
        // Write the HHL bands to the file
        file << "HHL" << std::endl;
        write_3d_vector(file, output, current_depth, 2 * current_depth, current_rows, 2 * current_rows, 0, current_cols);
        // Write the HHH bands to the file
        file << "HHH" << std::endl;
        write_3d_vector(file, output, current_depth, 2 * current_depth, current_rows, 2 * current_rows, current_cols, 2 * current_cols);
 

    // Close the file
    file.close();
}