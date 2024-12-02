#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-00:05:00
###SBATCH --reservation=cce3015

# job parameters
#SBATCH --output=/opt/users/gpel0001/cce3015/ssh/out/cuda_test_%A_%a.out
#SBATCH --error=/opt/users/gpel0001/cce3015/ssh/err/cuda_test_%A_%a.err
#SBATCH --job-name=cudatest
#SBATCH --account=undergrad

# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

echo Running on $(hostname)
scontrol --details show jobs $SLURM_JOBID | grep RES
env | grep CUDA
nvidia-smi

# Export CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Compile the CUDA program
nvcc -o /opt/users/gpel0001/cce3015/ssh/cudatest/cudatest_binary /opt/users/gpel0001/cce3015/ssh/cudatest/cudatest.cu

# Run the compiled CUDA program
/opt/users/gpel0001/cce3015/ssh/cudatest/cudatest_binary