#!/bin/bash

#SBATCH --nodes=15
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/DAMP/ECG/ECG8M.txt 8000000 64 128 ../results/discords/ECG8M_64_128_16cpu_K40.csv ../results/times/ECG8M_64_128_16cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512