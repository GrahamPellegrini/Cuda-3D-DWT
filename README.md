# CUDA 3D Discrete Wavelet Transform (3D-DWT)

## Overview

This project was developed as part of the **CCE3015 – High Performance Computing** course at the **University of Malta**, under the supervision of **Prof. Johann A. Briffa**.

The project’s objective was to explore and implement a **3D Discrete Wavelet Transform (3D-DWT)**, starting from a serial CPU-based approach and progressing to a CUDA-optimized parallel implementation using GPU acceleration.

The dataset used for the implementation and testing is from the **CHAOS Challenge**:
> *Kavur, A. E., et al. (2021). "CHAOS Challenge - Combined (CT-MR) Healthy Abdominal Organ Segmentation."*  
> Download available at: https://chaos.grand-challenge.org/Download/

### Course Project Phases

The project was conducted in three parts:
1. **Lab Exercises**: Foundation in CUDA and parallel computing.
2. **Serial Implementation**: A baseline 3D-DWT algorithm developed in C/C++.
3. **Parallel CUDA Implementation**: A GPU-accelerated version aiming to significantly reduce computation time.

> 📄 **Both implementations are accompanied by formal reports** located in the `docs/` folder, submitted as part of the unit’s coursework.

### Project Structure

```plaintext
.
├── data/               # Sample test data (CHAOS dataset samples)
├── docs/               # Project reports for serial and parallel implementations
├── include/            # Header files
├── src/                # Source code for both serial and CUDA implementations
├── Makefile            # Build configuration
└── README.md           # Project description (you are here)
