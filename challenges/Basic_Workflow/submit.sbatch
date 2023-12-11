#!/bin/bash 
#SBATCH -A PLS0144
#SBATCH -J add_vec_cpu
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

srun -n 1 --ntasks-per-node=1 ./run 
