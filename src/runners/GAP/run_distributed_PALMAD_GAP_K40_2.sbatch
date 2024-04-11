#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/VALMOD/GAP/GAP500K.txt 500000 128 256 ../results/discords/GAP500K_128_256_2cpu_K40.csv ../results/times/GAP500K_128_256_2cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512