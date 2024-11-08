#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-00:01:00
# job parameters
#SBATCH --output=/opt/users/gpel0001/cce3207/ssh/out/cce3207_slurm_%A_%a.out
#SBATCH --error=/opt/users/gpel0001/cce3207/ssh/err/cce3207_slurm_%A_%a.err
#SBATCH --job-name=cce3207_simpleCNN
#SBATCH --reservation=cce3207
#SBATCH --account=undergrad
# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all
#
VENV=/opt/users/gpel0001/cce3207/cce3207-venv
if [ -d $VENV ]; then
   echo Virtual environment found, activating
   VENV+=/bin/activate
   source "$VENV"
else
   echo Virtual environment not found!
fi

# Run your C++ binary with the desired arguments for assignment 1
assignment-1/bin/assignment-1 assignment-1/file/input.bin assignment-1/file/three_db2.bin 2 3