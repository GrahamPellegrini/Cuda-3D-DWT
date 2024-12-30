#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-00:05:00
#SBATCH --reservation=cce3015

# job parameters
#SBATCH --output=/opt/users/gpel0001/cce3015/ssh/out/make_clean.out
#SBATCH --error=/opt/users/gpel0001/cce3015/ssh/err/make_clean.err
#SBATCH --job-name=make 
#SBATCH --account=undergrad

# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

nvcc --version
which nvcc
# Directory of the makefile
cd /opt/users/gpel0001/cce3015/assignment-2

# Clean the project and check for failure
make clean || { echo "Make failed! Exiting." >&2; exit 1; }
