
void dwt_3d(float* volume, int depth, int rows, int cols) {
    dim3 block_dim(256); // Number of threads per block

    // Step 1: Row-wise DWT
    dim3 grid_dim_row((depth * rows + block_dim.x - 1) / block_dim.x); // Grid size for rows
    size_t shared_mem_row = cols * sizeof(float); // Shared memory size per block
    dwt_1d_kernel<<<grid_dim_row, block_dim, shared_mem_row>>>(volume, rows, cols, depth * rows, cols);
    cudaDeviceSynchronize();

    // Step 2: Column-wise DWT
    dim3 grid_dim_col((depth * cols + block_dim.x - 1) / block_dim.x); // Grid size for columns
    size_t shared_mem_col = rows * sizeof(float); // Shared memory size per block
    dwt_1d_kernel<<<grid_dim_col, block_dim, shared_mem_col>>>(volume, cols, rows, depth * cols, rows);
    cudaDeviceSynchronize();

    // Step 3: Depth-wise DWT
    dim3 grid_dim_depth((rows * cols + block_dim.x - 1) / block_dim.x); // Grid size for depth
    size_t shared_mem_depth = depth * sizeof(float); // Shared memory size per block
    dwt_1d_kernel<<<grid_dim_depth, block_dim, shared_mem_depth>>>(volume, rows * cols, 1, depth, 1);
    cudaDeviceSynchronize();
}




        else {
            int new_depth = (depth + 1) / 2;
            int new_rows = (rows + 1) / 2;
            int new_cols = (cols + 1) / 2;

            float* d_temp = nullptr;
            cudaError_t err = cudaMalloc(&d_temp, new_depth * new_rows * new_cols * sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate GPU memory for temporary volume: " << cudaGetErrorString(err) << std::endl;
                cudaFree(d_transformed);
                return;
            }

            dim3 block_dim(8, 8, 8);
            dim3 grid_dim((new_cols + block_dim.x - 1) / block_dim.x, (new_rows + block_dim.y - 1) / block_dim.y, (new_depth + block_dim.z - 1) / block_dim.z);

            reduce<<<grid_dim, block_dim>>>(d_transformed, d_temp, depth, rows, cols);
            cudaFree(d_volume);

            depth = new_depth;
            rows = new_rows;
            cols = new_cols;
            d_volume = d_temp;

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "Failed to synchronize the device: " << cudaGetErrorString(err) << std::endl;
                cudaFree(d_temp);
                return;
            }
            

            // Copy the transformed data back to the original volume
            copy_transformed_data<<<gridDim, blockDim>>>(d_data2, d_volume, depth, rows, cols, init_depth, init_rows, init_cols);
            // Synchronize the device
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "Failed to synchronize the device after copy_transformed_data: " << cudaGetErrorString(err) << std::endl;
            }



            
float* dwt_3d(float* d_volume, float* d_low_coeff, float* d_high_coeff, int depth, int rows, int cols, int filter_size) {
    float* d_data1 = d_volume;
    float* d_data2 = nullptr;
    cudaError_t err = cudaMalloc(&d_data2, depth * rows * cols * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for temporary volume: " << cudaGetErrorString(err) << std::endl;
    }

    dim3 blockDim(16, 8, 8);
    dim3 gridDim0((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    dim3 gridDim1((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    dim3 gridDim2((depth + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (rows + blockDim.z - 1) / blockDim.z);


    // Perform convolution along the first dimension
    dim0_kernel<<<gridDim0, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    // Synchronize the device
    cudaDeviceSynchronize();


    // Perform convolution along the second dimension
    dim1_kernel<<<gridDim1, blockDim, filter_size * sizeof(float) * 2>>>(d_data2, d_data1, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    // Synchronize the device  
    cudaDeviceSynchronize();

    // Perform convolution along the third dimension
    dim2_kernel<<<gridDim2, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    // Synchronize the device
    cudaDeviceSynchronize();

    return d_data2;
}