
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
