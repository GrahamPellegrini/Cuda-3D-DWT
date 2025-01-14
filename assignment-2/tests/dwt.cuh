#ifndef DWT3D_CUH
#define DWT3D_CUH

#include <vector>
#include <cuda_runtime.h>
#include "kernels.cuh"

// Define the wavelet coefficients as floats
extern const std::vector<std::vector<float>> db_low;
extern const std::vector<std::vector<float>> db_high;

// Forward declarations of functions
void toGPU(std::vector<std::vector<std::vector<float>>> volume, size_t db_num, size_t depth, size_t rows, size_t cols, size_t& filter_size, float*& d_volume);
std::vector<std::vector<std::vector<float>>> volCPU(float* d_volume, size_t depth, size_t rows, size_t cols);
void dwt_3d(float* d_volume, size_t depth, size_t rows, size_t cols, size_t filter_size);

#endif // DWT3D_CUH
