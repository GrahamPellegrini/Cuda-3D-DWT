#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --reservation=cce3015
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-00:01:00
# job parameters
#SBATCH --job-name=gputest
#SBATCH --account=undergrad
# email user with progress
#SBATCH --mail-user=name.surname@um.edu.mt
#SBATCH --mail-type=all
#
echo Running on $(hostname)
scontrol --details show jobs $SLURM_JOBID |grep RES
env | grep CUDA
nvidia-smi
./testcuda/bin/cudatest
