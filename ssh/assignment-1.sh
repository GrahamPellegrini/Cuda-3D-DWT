#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-00:05:00
##SBATCH --reservation=cce3015

# job parameters
#SBATCH --output=/opt/users/gpel0001/cce3015/ssh/out/assignment_1.out
#SBATCH --error=/opt/users/gpel0001/cce3015/ssh/err/assignment_1.err
#SBATCH --job-name=cce3015_assignment-1
#SBATCH --account=undergrad

# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

# Directory of the assignment 1
cd /opt/users/gpel0001/cce3015/assignment-1

# Clean old binaries and check for failure
make clean
# Give the makefile time to clean the project
sleep 1
# Make the project
make
# Give the makefile time to finish making binaries
sleep 5
# Run your C++ binary with the desired arguments for assignment 1
./bin/assignment-1 file/input.bin file/single_haar.bin 1 1 