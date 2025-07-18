<h1 align="center">CUDA 3D Discrete Wavelet Transform (3D-DWT)</h1>

<p align="center">
  <a href="https://www.um.edu.mt/courses/studyunit/CCE3015">
    <img src="https://img.shields.io/badge/University%20of%20Malta-CCE3015-blue?style=for-the-badge&logo=nvidia&logoColor=white" alt="CCE3015 Course">
  </a>
  <a href="https://developer.nvidia.com/cuda-zone">
    <img src="https://img.shields.io/badge/Built%20with-CUDA%20C%2B%2B-green?style=for-the-badge&logo=nvidia" alt="CUDA">
  </a>
  <a href="https://chaos.grand-challenge.org/">
    <img src="https://img.shields.io/badge/Dataset-CHAOS%20Challenge-red?style=for-the-badge" alt="CHAOS Dataset">
  </a>
</p>

<p align="center">
  Parallelized GPU implementation of the 3D Discrete Wavelet Transform (3D-DWT) using CUDA C++, built for high-performance volume data processing and benchmarking against serial DWT.
</p>

---

## Project Overview

This repository presents both a **serial and parallel CUDA implementation** of the **3D Discrete Wavelet Transform (3D-DWT)** developed for the study unit [**CCE3015 – High Performance Computing**](https://www.um.edu.mt/courses/studyunit/CCE3015) at the [**University of Malta**](https://www.um.edu.mt/), under the supervision of [**Prof. Johann A. Briffa**](https://www.um.edu.mt/profile/johannbriffa).

The project aims to transform large 3D volumetric datasets (e.g., medical CT/MRI scans) using the DWT, a foundational technique in signal processing. 

While the **serial version** includes a complete **multi-level DWT** and its inverse, the **CUDA implementation focuses on a single-level DWT**, optimized for GPU acceleration due to time and resource constraints.

Documentation and reports for both implementations are included in the `docs/` and `latex/` directories.


## Serial Implementation

The serial implementation performs a complete **multi-level 3D Discrete Wavelet Transform (DWT)** and its **inverse**, designed as the foundation for later CUDA parallelization. It processes medical imaging volumes by recursively applying 1D wavelet transforms along each axis (columns → rows → depth), generating eight subbands per level (e.g. LLL, LLH, ..., HHH).

### Key Steps:
- **Preprocessing**: DICOM slices are converted into NumPy 3D arrays in Python and saved as binary `.bin` files, making them compatible with C++ I/O routines.
- **Filter Selection**: Daubechies wavelets (db1–db4) are supported using hardcoded low-pass and high-pass coefficients, sourced from PyWavelets for validation.
- **1D DWT**: A core function convolves 1D signals with each filter, downsamples the result, and separates approximation and detail coefficients.
- **3D DWT**: The 1D DWT is applied in sequence across the three dimensions. Each axis processes all volume slices using nested loops, storing back the coefficients in-place to minimize memory usage.
- **Multi-Level Decomposition**: A recursive wrapper calls the 3D DWT on the LLL (approximation) subband, halving dimensions each time. Padding is applied as needed for odd dimensions.
- **Inverse DWT**: The reconstruction reverses the transform, validating the correctness of the subband structure and ensuring loss is within expected tolerances.

### Evaluation:
- Python scripts post-process the output `.bin` files and compare the subbands against reference results from PyWavelets using **MSE** and **Euclidean distance**.
- The implementation was benchmarked on datasets of varying resolution, showing strong consistency across levels and filters.

The serial version was carefully structured for performance and clarity, using modular C++ and a portable Makefile setup. It serves as the performance baseline and functional reference for the CUDA implementation.

SLURM was used to submit batch jobs to the university GPU cluster using `sbatch`.


## Core Parallelization Objective

Assignment 2 builds upon the serial implementation by porting the DWT algorithm to **CUDA**, with the goal of significantly reducing execution time for large 3D volumes. Rather than modifying the serial algorithm directly, the code was refactored to fit CUDA's programming model and memory hierarchy.

The key objectives of the parallelization phase were:

- **Separating computation by dimension**: Independent CUDA kernels were developed to apply the 1D DWT along the X, Y, and Z axes. These were launched in sequence to respect data dependencies while maximizing per-axis parallelism.
- **Memory layout and management**: The 3D volume was flattened into a contiguous 1D array (row-major order) to ensure compatibility with GPU memory and minimize overhead during transfers.
- **Optimizing filter access**: Both shared memory and constant memory were tested for storing wavelet coefficients. Ultimately, constant memory yielded better performance due to better cache behavior and less block-level replication.
- **Swapping buffers efficiently**: Instead of copying outputs between levels, input/output buffers were alternated after each kernel call, minimizing memory overhead and transfer time.
- **Profiler-driven development**: `cudaEvent`, Nsight, and command-line profilers were used to measure kernel performance and identify bottlenecks early in development.

Due to the inherent complexity of recursive data dependency in multi-level DWT, this implementation is limited to a **single-level transform**. However, it forms a modular and well-optimized baseline from which full multi-level parallel DWT can be extended in future work.


## Folder Structure (Parallel)

```
.
├── Parallel/
│   ├── data/              # Input and output binary volume files
│   ├── docs/              # Assignment 2 report
│   ├── include/           # Header files for CUDA kernels and utilities
│   ├── src/               # Main CUDA implementation
│   │   ├── assignment-2.cu
│   │   ├── kernels.cuh
│   │   ├── idwt.h
│   │   ├── loadbin.h, savebin.h
│   ├── latex/             # LaTeX source for report
│   ├── Makefile           # Build system (debug, release, profiler targets)
└── README.md
```


## CUDA Implementation Highlights

- DWT is applied dimension-wise with **separate CUDA kernels** for X, Y, and Z axes.
- Memory is flattened in **row-major order** and transferred to the GPU.
- **Grid/block sizes** dynamically adjust to input volume dimensions.
- Input/output buffers are **swapped** between kernels to avoid redundant transfers.
- Evaluated use of **shared memory vs constant memory** for filter coefficients.

### Memory Optimization Comparison

| Memory Type     | Transfer Time | Kernel Time | Total Time  |
|-----------------|---------------|-------------|-------------|
| Shared Memory   | 0.856 ms      | 12.439 ms   | 13.295 ms   |
| Constant Memory | 7.369 ms      | 4.427 ms    | 11.796 ms   |

Constant memory resulted in ~11% overall speedup due to improved caching and reduced kernel execution time.


## Performance Evaluation

| Dataset             | Serial Time | Parallel Time | Speedup  |
|---------------------|-------------|----------------|----------|
| 512×512×78 (large) | 170.77 ms  | 76.81 ms       | 2.22x    |
| 128×128×20 (small) | 6.99 ms    | 7.81 ms        | 0.89x    |

CUDA acceleration proves effective for large datasets where the cost of memory transfers is amortized.


## Build Instructions

### Requirements
- CUDA Toolkit (v11.0 or later)
- GCC/G++ compiler
- NVIDIA GPU (Compute Capability 5.0+)

### Compile
From project root:
```bash
cd Parallel
make release         # Optimized build
make debug           # Debug build
make nsys / make ncu # For profiling builds
```


## Run Instructions

### Serial (for reference)
```bash
./bin/assignment-1 <input_volume_file> <output_volume_file> <db_num> <levels>
```
- `db_num`: Index of the Daubechies wavelet filter
- `levels`: Number of DWT decomposition levels to apply

### CUDA (Parallel)
```bash
./bin/assignment-2 <input_volume_file> <output_volume_file> <db_num> <inverse_flag>
```
- `db_num`: Index of the Daubechies wavelet filter
- `inverse_flag`: `0` for forward transform, `1` for inverse (serial fallback only)

SLURM was used with `sbatch` to submit batch jobs to the GPU cluster.


## Author

**Graham Pellegrini**  
B.Eng. (Hons) Computer Engineering  
University of Malta  
GitHub: [@GrahamPellegrini](https://github.com/GrahamPellegrini)
