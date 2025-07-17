# CUDA 3D Discrete Wavelet Transform (3D-DWT)

## Overview

This project was developed as part of the **CCE3015 – High Performance Computing** course at the **University of Malta**, under the supervision of **Prof. Johann A. Briffa**. It represents the second phase of a two-part assignment, focusing on GPU-accelerated parallelization of a 3D Discrete Wavelet Transform (3D-DWT) using CUDA.

The implementation uses sample data from the **CHAOS Challenge** dataset:

> *Kavur, A. E., et al. (2021). "CHAOS Challenge - Combined (CT-MR) Healthy Abdominal Organ Segmentation."*  
> Download available at: [CHAOS Challenge](https://chaos.grand-challenge.org/Download/)

---

## Project Summary

This project adapts a serial 3D-DWT algorithm into a CUDA-parallelized version. Due to time constraints, only **single-level decomposition** is implemented, with a strong emphasis on GPU optimization and memory management. Multi-level DWT and inverse transforms are included in the serial version but not parallelized.

---

## Key Features

- ✅ CUDA-based implementation of 3D Discrete Wavelet Transform (DWT)
- 📦 Support for different 3D volume sizes (e.g., medical CT scan volumes)
- 🧠 Memory optimization using **constant vs shared memory** strategies
- 📊 Performance profiling using **NVIDIA Nsight and `cudaEvent`**
- 📁 Reports documenting both serial and parallel approaches included in `docs/`

---

## Project Structure

```
.
├── data/               # Input volume data (CHAOS dataset samples)
├── docs/               # Detailed reports for both implementations
├── include/            # Header files: DWT kernels, I/O, and inverse transform
├── src/                # Main CUDA source file and supporting kernel logic
├── Makefile            # Build system (debug/release/profiling targets)
└── README.md           # You are here!
```

---

## Parallel Implementation Details

- The DWT is applied **dimension-wise** (X → Y → Z) using **separate CUDA kernels** for each axis.
- **Block size** of `(16, 16, 4)` was chosen based on volume shape (rows/cols >> depth).
- **Grid dimensions** are dynamically calculated to ensure full 3D coverage.
- **Memory access optimizations**:
  - Early attempts with shared memory were replaced by **constant memory**, which yielded lower execution times and better cache utilization.
  - A packed coefficient array reduced `cudaMemcpyToSymbol` overhead.
- **Input/output buffers are swapped** between kernel calls to minimize unnecessary data transfers.
- Volumes are **flattened** using row-major order for GPU memory compatibility and restored post-processing.

---

## Build Instructions

### Requirements

- NVIDIA GPU (Compute Capability 5.0+)
- CUDA Toolkit (v11.0+)
- GCC/G++ compiler and `make`

### Build

From the project root:

```bash
cd src
make release
```

Other targets:
- `make debug` – Enables assertions and profiling
- `make ncu` / `make nsys` – Integration with NVIDIA profiling tools

---

## Run Instructions

```bash
./cuda_3d_dwt <input_volume_file>
```

- The input should be a 3D volume stored as a binary file.
- The output will be a forward-transformed volume.
- I/O is handled via `loadbin.h` and `savebin.h` with minimal overhead.

---

## Performance Highlights

### Shared vs Constant Memory
| Memory Type      | Transfer Time | Kernel Time | Total       |
|------------------|---------------|-------------|-------------|
| Shared Memory    | 0.856 ms      | 12.439 ms   | 13.295 ms   |
| Constant Memory  | 7.369 ms      | 4.427 ms    | 11.796 ms   |

✅ Constant memory yielded ~**11% speedup** due to reduced kernel execution latency.

### Speedup Over Serial

| Dataset              | Serial Time | Parallel Time | Speedup  |
|----------------------|-------------|----------------|----------|
| Large (512×512×78)   | 170.77 ms   | 76.81 ms       | **2.22×** |
| Small (128×128×20)   | 6.99 ms     | 7.81 ms        | **0.89×** (slower due to memory overhead)

> 🚀 CUDA acceleration is more effective with larger datasets where memory transfer cost is amortized.

---

## Challenges & Lessons Learned

- Handling **memory layout, flattening, and indexing** across dimensions was complex.
- **Block and grid design** was critical for volume alignment and optimal thread usage.
- **Shared memory** performed worse than **constant memory** due to replication overhead.
- **Multi-level DWT** proved too complex for the CUDA scope in this assignment but is viable for future work.

---

## License

This project was developed for academic purposes under the CCE3015 course.  
The CHAOS dataset is credited to its original authors and is subject to their [terms of use](https://chaos.grand-challenge.org/Download/).

---

## Author

**Graham Pellegrini**  
B.Eng. Computer Engineering — University of Malta  
GitHub: [@GrahamPellegrini](https://github.com/GrahamPellegrini)
