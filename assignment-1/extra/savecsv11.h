#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

// Function to save the 3D volume to a CSV file
void save_csv(const std::string& filename, 
              const std::vector<std::vector<std::vector<float>>>& volume, 
              int depth, int rows, int cols, int multi_level) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Write the dimensions and multi_level as the first line
    file << depth << "," << rows << "," << cols << "," << multi_level << "\n";

    // Write the volume data
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                file << volume[d][r][c];
                if (c < cols - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
    }

    file.close();
}