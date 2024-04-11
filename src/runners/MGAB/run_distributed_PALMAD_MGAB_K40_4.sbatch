#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/DAMP/MGAB.csv 100000 64 128 ../results/discords/MGAB_64_128_4cpu_K40.csv ../results/times/MGAB_64_128_4cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512