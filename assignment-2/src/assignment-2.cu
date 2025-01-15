// Cuda kernels for the 3D DWT
#include "../include/kernels.cuh"

// Inverse multi-level DWT serial implementation
#include "../include/idwt.h"
// Note this include contains all other necessary includes, even the load and bin headers

// Include for size_t type
#include <cstddef>


// Define the wavelet coefficients as 2D vector floats
// Low coefficients
const std::vector<std::vector<float>> db_low = {
    {0.70710678f, 0.70710678f}, // db1
    {-0.12940952f, 0.22414387f, 0.83651630f, 0.48296291f}, // db2
    {0.03522629f, -0.08544127f, -0.13501102f, 0.45987750f, 0.80689151f, 0.33267055f}, // db3
    {-0.01059740f, 0.03288301f, 0.03084138f, -0.18703481f, -0.02798377f, 0.63088077f, 0.71484657f, 0.23037781f} // db4
};
// High coefficients
const std::vector<std::vector<float>> db_high = {
    {-0.70710678f, 0.70710678f}, // db1
    {-0.48296291f, 0.83651630f, -0.22414387f, -0.12940952f}, // db2
    {-0.33267055f, 0.80689151f, -0.45987750f, -0.13501102f, 0.08544127f, 0.03522629f}, // db3
    {-0.23037781f, 0.71484657f, -0.63088077f, -0.02798377f, 0.18703481f, 0.03084138f, -0.03288301f, -0.01059740f} // db4
};


// Function to handle the GPU memory allocation and data transfer
void toGPU(std::vector<std::vector<std::vector<float>>> volume, size_t db_num, size_t depth, size_t rows, size_t cols, size_t& filter_size, float*& d_volume) 
{
    // Select the coefficients based on db_num
    std::vector<float> low_coeff = db_low[db_num - 1];
    std::vector<float> high_coeff = db_high[db_num - 1];

    // Calculate the filter size 
    filter_size = low_coeff.size();
    assert(filter_size < MAX_FILTER_SIZE && "Filter size exceeds constant memory capacity");
    
    // Pack coefficients
    std::vector<float> combined_coeff(filter_size * 2);
    std::copy(low_coeff.begin(), low_coeff.end(), combined_coeff.begin());
    std::copy(high_coeff.begin(), high_coeff.end(), combined_coeff.begin() + filter_size);

    // Make sure the data is aligned in memory
    assert(reinterpret_cast<uintptr_t>(combined_coeff.data()) % 16 == 0 && "Data is not 16-byte aligned");

    #if DEBUG
       
        // Make a cuda event to calculate the time taken by the mem copy
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
    #endif

    cudaError_t err = cudaMemcpyToSymbol(d_coeff, combined_coeff.data(), filter_size * 2 * sizeof(float));
    assert(err == cudaSuccess && "Failed to copy coefficients to constant memory");

    #if DEBUG
        // Record the stop event and synchronize
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        // Print the time taken for the mem copy (Not DEBUG)
        std::cerr << "Time taken for copying coefficients to constant memory: " << milliseconds << "ms" << std::endl;
    #endif

    // Flatten the 3D volume into a 1D vector (row-major order)
    std::vector<float> flat_volume(depth * rows * cols);
    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                flat_volume[d * rows * cols + r * cols + c] = volume[d][r][c];
            }
        }
    }

    // Allocate memory on the GPU for the volume
    err = cudaMalloc(&d_volume, flat_volume.size() * sizeof(float));
    assert(err == cudaSuccess && "Failed to allocate GPU memory for the volume");

    // Copy the flattened volume to the GPU
    err = cudaMemcpy(d_volume, flat_volume.data(), flat_volume.size() * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess && "Failed to copy flattened volume to GPU");

    // Clear the CPU memory after copying to GPU
    flat_volume.clear();
    volume.clear();
}

// Function to copy the transformed volume from the GPU back to the CPU
std::vector<std::vector<std::vector<float>>> volCPU(float* d_volume, size_t depth, size_t rows, size_t cols)
{
    // Allocate memory for the 3D volume on the CPU 
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    // Make a 1D vector to store the flattened volume
    std::vector<float> flat_volume(depth * rows * cols);
    // Copy the flattened volume from the GPU to the CPU
    cudaError_t err = cudaMemcpy(flat_volume.data(), d_volume, flat_volume.size() * sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "Failed to copy flattened volume from GPU to CPU");

    // Unflatten the 1D vector into a 3D volume 
    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                volume[d][r][c] = flat_volume[(d * rows * cols) + (r * cols) + c];
            }
        }
    }

    // Free the allocated GPU memory
    err = cudaFree(d_volume);
    assert(err == cudaSuccess && "Failed to free the GPU memory from the volume");

    return volume;
}

// Function to perform the 3D DWT on the GPU using the CUDA kernels
void dwt_3d(float* d_volume, size_t depth, size_t rows, size_t cols, size_t filter_size) 
{
    // Cuda event used to measure the time taken for the DWT
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Set and allocate the temporary data pointers
    float* d_data1 = d_volume;
    float* d_data2 = nullptr;
    cudaError_t err = cudaMalloc(&d_data2, depth * rows * cols * sizeof(float));
    assert(err == cudaSuccess && "Failed to allocate GPU memory for temporary data2");

    // Set the block dimensions for the kernel
    dim3 blockDim(16, 16, 4); 

    // Define and calculate the grid dimensions for the kernels
    dim3 row_grid((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    dim3 col_grid((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    dim3 depth_grid((depth + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (rows + blockDim.z - 1) / blockDim.z);
    dim3 map_grid((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);

    // Perform convolution along the rows
    row_kernel<<<row_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, filter_size, depth, rows, cols);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "Failed to synchronize the GPU threads at row kernel");

    // Perform convolution along the columns
    col_kernel<<<col_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data2, d_data1, filter_size, depth, rows, cols);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "Failed to synchronize the GPU threads at column kernel");

    // Perform convolution along the depth
    depth_kernel<<<depth_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, filter_size, depth, rows, cols);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "Failed to synchronize the GPU threads at depth kernel");

    // Map the transformed data back to the original volume
    map_kernel<<<map_grid, blockDim>>>(d_volume, d_data2, depth, rows, cols);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "Failed to synchronize the GPU threads at map kernel");

    // Free the temporary volume
    err = cudaFree(d_data2);
    assert(err == cudaSuccess && "Failed to free the GPU memory from the temporary data2");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Stop and print the time taken for the DWT (Not DEBUG)
    std::cerr << "Time taken for DWT: " << milliseconds << "ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


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

