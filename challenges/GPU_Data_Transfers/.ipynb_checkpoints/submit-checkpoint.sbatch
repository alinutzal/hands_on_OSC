#!/bin/bash 
#SBATCH -A PLS0144
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH -J add_vec_cpu
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch

module load cuda

srun --gpus=1 ./run 
