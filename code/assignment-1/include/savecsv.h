#ifndef SAVECSV_H
#define SAVECSV_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void save_coefficients_to_csv(const std::string& csv_file,
                              const std::vector<std::vector<std::vector<float>>>& LLL,
                              const std::vector<std::vector<std::vector<float>>>& LLH,
                              const std::vector<std::vector<std::vector<float>>>& LHL,
                              const std::vector<std::vector<std::vector<float>>>& LHH,
                              const std::vector<std::vector<std::vector<float>>>& HLL,
                              const std::vector<std::vector<std::vector<float>>>& HLH,
                              const std::vector<std::vector<std::vector<float>>>& HHL,
                              const std::vector<std::vector<std::vector<float>>>& HHH) {
    std::ofstream file(csv_file);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << csv_file << std::endl;
        return;
    }
    
    // Write the shape of the volume
    int rows = LLL.size();
    int cols = LLL[0].size();
    int depth = LLL[0][0].size();
    file << rows << " " << cols << " " << depth << std::endl;
    
    // Helper lambda to write a 3D vector to the file
    auto write_3d_vector = [&file](const std::vector<std::vector<std::vector<float>>>& vec) {
        for (const auto& slice : vec) {
            for (const auto& row : slice) {
                for (const auto& val : row) {
                    file << val << " ";
                }
                file << std::endl;
            }
        }
    };
    
    // Write each sub-band to the file
    file << "LLL" << std::endl;
    write_3d_vector(LLL);
    file << "LLH" << std::endl;
    write_3d_vector(LLH);
    file << "LHL" << std::endl;
    write_3d_vector(LHL);
    file << "LHH" << std::endl;
    write_3d_vector(LHH);
    file << "HLL" << std::endl;
    write_3d_vector(HLL);
    file << "HLH" << std::endl;
    write_3d_vector(HLH);
    file << "HHL" << std::endl;
    write_3d_vector(HHL);
    file << "HHH" << std::endl;
    write_3d_vector(HHH);
    
    file.close();
}

#endif // SAVECSV_H