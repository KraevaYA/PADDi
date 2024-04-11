#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/DAMP/RandomWalk/RandomWalk500K.txt 500000 64 128 ../results/discords/RandomWalk500K_64_128_2cpu_K40.csv ../results/times/RandomWalk500K_64_128_2cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512