#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/DAMP/ECG/ECG250K.txt 250000 64 128 ../results/discords/ECG250K_64_128_8cpu_K40.csv ../results/times/ECG250K_64_128_8cpu_K40.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512