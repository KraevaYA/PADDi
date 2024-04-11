#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/DAMP/bearing.txt 244189 128 256 ../results/discords/bearing_128_256_8cpu_K40.csv ../results/times/bearing_128_256_8cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512