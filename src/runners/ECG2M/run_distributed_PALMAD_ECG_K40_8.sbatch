#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/DAMP/ECG/ECG2M.txt 2000000 64 128 ../results/discords/ECG2M_64_128_8cpu_K40.csv ../results/times/ECG2M_64_128_8cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512