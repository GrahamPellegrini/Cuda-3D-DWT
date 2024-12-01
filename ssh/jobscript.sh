#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-00:05:00
# job parameters
#SBATCH --output=/opt/users/gpel0001/cce3015/ssh/out/cce3015_slurm_%A_%a.out
#SBATCH --error=/opt/users/gpel0001/cce3015/ssh/err/cce3015_slurm_%A_%a.err
#SBATCH --job-name=cudatest
#SBATCH --account=undergrad
# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

echo Running on $(hostname)
scontrol --details show jobs $SLURM_JOBID |grep RES
env | grep CUDA
nvidia-smi

nvcc -o ./ssh/cudatest ./ssh/cudatest.cu
./ssh/cudatest