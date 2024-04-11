#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/VALMOD/GAP/GAP1M.txt 1000000 128 256 ../results/discords/GAP1M_128_256_4cpu_K40.csv ../results/times/GAP1M_128_256_4cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512