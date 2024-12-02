#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-00:05:00
#SBATCH --reservation=cce3015

# job parameters
#SBATCH --output=/opt/users/gpel0001/cce3015/ssh/out/assignment_2_%A_%a.out
#SBATCH --error=/opt/users/gpel0001/cce3015/ssh/err/assignment_2_%A_%a.err
#SBATCH --job-name=cce3015_assignment-1
#SBATCH --account=undergrad


# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

# Export CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

VENV=/opt/users/gpel0001/cce3015/cce3015-venv
if [ -d $VENV ]; then
   echo Virtual environment found, activating
   VENV+=/bin/activate
   source "$VENV"
else
   echo Virtual environment not found!
fi

# Run your C++ binary with the desired arguments for assignment 1
assignment-2/bin/assignment-2 assignment-2/file/input.bin assignment-2/file/three_db2.bin 2 3