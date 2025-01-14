// Custom headers
#include "../include/loadbin.h"
#include "../include/savebin.h"
#include "../include/dwt.cuh"
#include "../include/idwt.h"

// Non-inherited icnludes



// Main program entry point
int main(int argc, char *argv[]) {
    (void)argc; // Suppress unused parameter warning
    // Print the program title (DEBUG)
    #ifdef DEBUG
        std::cerr << "Assignment 2: CUDA Implementation of 3D DWT" << std::endl;
    #endif

    // Check if the number of arguments is correct
    assert(argc == 5 && "Usage: ./assignment-2 <input.bin> <output.bin> <db_num> <inverse>");

    // Load the arguments into variables
    std::string bin_in = argv[1];
    std::string bin_out = argv[2];
    size_t db_num = std::stoi(argv[3]);
    size_t inverse = std::stoi(argv[4]);

    // Load the 3D volume from the binary file
    std::vector<std::vector<std::vector<float>>> vol_in = loadvolume(bin_in);

    // Get the dimensions of the 3D volume
    size_t depth = vol_in.size();
    size_t rows = vol_in[0].size();
    size_t cols = vol_in[0][0].size();

    // Print the dimensions of the volume before any processing (DEBUG)
    #ifdef DEBUG
        std::cerr << "Volume dimension in : " << depth << "x" << rows << "x" << cols << std::endl;
    #endif

    // Define volume for the output
    std::vector<std::vector<std::vector<float>>> vol_out;

    // Perform the inverse if inverse flag is set to 1
    if (inverse == 1) {
        // Notice that the inverse is being performed (DEBUG)
        #ifdef DEBUG
            std::cerr << "Performing inverse DWT" << std::endl;
        #endif

        // Perform the inverse DWT using the serial implementation (CPU)
        inverse_multi_level(vol_in, db_num, 1);

        // Swap vol_in and name it vol_out
        vol_out = vol_in;

        // Check if vol_out is empty
        assert(!vol_out.empty() && "Volume out is empty");

        // Print the dimensions of volume after inverse DWT (DEBUG)
        #ifdef DEBUG
            std::cerr << "Volume dimensions out (" << db_num << "db IDWT): " << vol_out.size() << "x" << vol_out[0].size() << "x" << vol_out[0][0].size() << std::endl;
        #endif
    }

    else{
        
        // Create pointers for the volume on the GPU 
        float* d_volume = nullptr;
        // Indicator of the filters sizes to be iterated over
        size_t filter_size;

        // Copy the db Coeffs to constant memory and Volume to global memory on the GPU
        toGPU(vol_in, db_num, depth, rows, cols, filter_size, d_volume);
        
        // Perform the Cuda 3D DWT in the GPU
        dwt_3d(d_volume, depth, rows, cols, filter_size);

        // Copy the transformed volume back to the CPU
        vol_out = volCPU(d_volume, depth, rows, cols);

        // Check if vol_out is empty
        assert(!vol_out.empty() && "Volume out is empty");

        // Print the dimensions of volume after DWT (DEBUG)
        #ifdef DEBUG
            std::cerr << "Volume dimensions out (" << db_num << "db DWT): " << vol_out.size() << "x" << vol_out[0].size() << "x" << vol_out[0][0].size() << std::endl;
        #endif
    }

    // Save the modified 3D volume to the output binary file
    savevolume(vol_out, bin_out);

    // Print seperator (DEBUG)
    #ifdef DEBUG
        std::cerr << "----------------------------------------" << std::endl;
        std::cerr << std::endl;
    #endif

    return EXIT_SUCCESS;
}
