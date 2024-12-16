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
#SBATCH --output=/opt/users/gpel0001/cce3015/ssh/out/make_%A_%a.out
#SBATCH --error=/opt/users/gpel0001/cce3015/ssh/err/make_%A_%a.err
#SBATCH --job-name=make 
#SBATCH --account=undergrad

# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

# make command 
cd /opt/users/gpel0001/cce3015/assignment-2
make