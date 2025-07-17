# CUDA 3D Discrete Wavelet Transform (3D-DWT)

## Project Overview

This repository presents both a **serial and parallel CUDA implementation** of the **3D Discrete Wavelet Transform (3D-DWT)** developed for the study unit [**CCE3015 – High Performance Computing**](https://www.um.edu.mt/courses/studyunit/CCE3015) at the [**University of Malta**](https://www.um.edu.mt/), under the supervision of [**Prof. Johann A. Briffa**](https://www.um.edu.mt/profile/johannbriffa).

The project aims to transform large 3D volumetric datasets (e.g., medical CT/MRI scans) using the DWT, a foundational technique in signal processing. 

While the **serial version** includes a complete **multi-level DWT** and its inverse, the **CUDA implementation focuses on a single-level DWT**, optimized for GPU acceleration due to time and resource constraints.

Documentation and reports for both implementations are included in the `docs/` and `latex/` directories.


## Serial Implementation

The serial implementation performs a full **multi-level 3D-DWT** and its **inverse transform**. It processes the input volume dimension-wise (X, Y, Z), recursively transforming the `LLL` sub-band at each level.

- Developed for **Assignment 1** of the course.
- Core logic resides in a separate repository.
- Tested and benchmarked using a subset of the **CHAOS Challenge** dataset.
- Generates outputs such as forward and inverse transformed volumes.

SLURM was used to submit batch jobs to the university GPU cluster using `sbatch`.


## Core Parallelization Objective

Assignment 2 builds upon the serial version by porting the DWT algorithm to **CUDA**, aiming to leverage parallel GPU processing for high-volume datasets. The parallel implementation includes:

- CUDA kernel development for X, Y, Z axis transforms.
- Flattened memory layout for efficient GPU access.
- Evaluation of **shared vs constant memory** strategies.
- Profiling with `cudaEvent`, Nsight, and NVIDIA CLI tools.

Due to complexity in managing sub-bands and dependencies, the CUDA version was scoped to a **single-level DWT**. However, it lays a strong foundation for future extension to full multi-level transforms.


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


## Dataset

The CHAOS dataset was used for performance testing:

> Kavur, A. E., et al. (2021). "CHAOS Challenge - Combined (CT-MR) Healthy Abdominal Organ Segmentation."  
> [https://chaos.grand-challenge.org/Download/](https://chaos.grand-challenge.org/Download/)


## Acknowledgements

- [University of Malta – Faculty of ICT](https://www.um.edu.mt/ict)
- [Prof. Johann A. Briffa – Lecturer & Supervisor](https://www.um.edu.mt/profile/johannbriffa)
- [CHAOS Challenge Dataset](https://chaos.grand-challenge.org/Download/)
- [GitHub Copilot](https://github.com/features/copilot) (for minor code suggestion assistance)


## Author

**Graham Pellegrini**  
B.Eng. (Hons) Computer Engineering  
University of Malta  
GitHub: [@GrahamPellegrini](https://github.com/GrahamPellegrini)
