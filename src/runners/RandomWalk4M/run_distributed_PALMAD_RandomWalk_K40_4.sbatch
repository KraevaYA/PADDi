#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/DAMP/RandomWalk/RandomWalk4M.txt 4000000 128 256 ../results/discords/RandomWalk4M_128_256_4cpu_K40.csv ../results/times/RandomWalk4M_128_256_4cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512