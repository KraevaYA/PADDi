#!/bin/bash

#SBATCH --nodes=15
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/VALMOD/GAP/GAP2M.txt 2000000 128 256 ../results/discords/GAP2M_128_256_16cpu_K40.csv ../results/times/GAP2M_128_256_16cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512