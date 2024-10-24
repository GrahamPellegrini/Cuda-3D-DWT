#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <cstdlib> // For std::exit
#include "ct3d.h"  // Include your 3D image header
#include "../shared/jbutil.h" // Include the jbuilt library

// Function to perform 1D DWT on a single row or column
void dwt_1d(const float* input, int length)
{
    // Check if the input length is a power of 2
    if (length & (length - 1)) {
        std::cerr << "Input length is not a power of 2" << std::endl;
        return; // Exit if the input length is not a power of 2
    }

    // Allocate memory for the output array
    float* output = new float[length];

    // Perform the DWT by iterating over the input array
    for (int i = 0; i < length; i += 2) {
        // Calculate the average and difference of the two input values
        // Average in the lower half
        output[i / 2] = (input[i] + input[i + 1]) / 2.0f; 
        // Difference in the upper half
        output[length / 2 + i / 2] = (input[i] - input[i + 1]) / 2.0f; 
    }

    // Copy the output back to the input array
    std::copy(output, output + length, const_cast<float*>(input));

    // Free the allocated memory
    delete[] output;
    return;
}

// Function to perform 2D DWT on a 2D image
void dwt_2d(jbutil::image<float>& image, jbutil::image<float>& LL, jbutil::image<float>& LH, jbutil::image<float>& HL, jbutil::image<float>& HH)
{}


// Processing function to load the DICOM images and perform the DWT
template <class T>
void process(const std::string& directory, int depth, float scale, float translation)
{
    // Create a 3D image instance without specifying rows and cols
    image3D<float> ctImage;

    // Load the DICOM images into the 3D image structure
    if (!ctImage.load_directory(directory, depth)) {
        std::cerr << "Failed to load images from directory: " << directory << std::endl;
        return; // Exit if loading fails
    }

    // Take a single slice from the 3D image
    jbutil::image<float> slice = ctImage.get_slice(0);

    // Create the output images for the DWT coefficients
    jbutil::image<float> LL(slice.get_rows(), slice.get_cols(), 1, 0.0f);
    jbutil::image<float> LH(slice.get_rows(), slice.get_cols(), 1, 0.0f);
    jbutil::image<float> HL(slice.get_rows(), slice.get_cols(), 1, 0.0f);
    jbutil::image<float> HH(slice.get_rows(), slice.get_cols(), 1, 0.0f);

    // Perform the 2D DWT on the slice
    dwt_2d(slice, LL, LH, HL, HH);

    // Save the DWT coefficients to disk
    LL.save("LL.dcm");


}

// Main program entry point
int main(int argc, char *argv[])
{
    std::cerr << "Assignment 1: Synchronous DWT on 3D CT Image" << std::endl;

    if (argc != 5) { // Expecting two arguments: directory and depth
        std::cerr << "Usage: " << argv[0] << " <directory> <depth> <scale> <translation>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string directory = argv[1];
    int depth = std::atoi(argv[2]); // Convert depth argument to integer
    float scale = std::atoi(argv[3]); // Convert scale argument to integer
    float translation = std::atoi(argv[4]); // Convert translation argument to integer

    // Process the 3D DWT
    process<float>(directory, depth, scale, translation);

    return EXIT_SUCCESS;
}
