#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=100M
#SBATCH --time=0-01:05:00
#SBATCH --reservation=cce3015

# job parameters
#SBATCH --output=/opt/users/gpel0001/cce3015/ssh/out/assignment-2.out
#SBATCH --error=/opt/users/gpel0001/cce3015/ssh/err/assignment-2.err
#SBATCH --job-name=cuda_2
#SBATCH --account=undergrad


# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

# Run your C++ binary with the desired arguments for assignment 1
assignment-2/bin/assignment-2 assignment-2/file/file_in.bin assignment-2/file/single_haar.bin 1 1