#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
 
mpirun ./main ../datasets/VALMOD/GAP/GAP250K.txt 250000 128 256 ../results/discords/GAP250K_128_256_1cpu_P100.csv ../results/times/GAP250K_128_256_1cpu_P100.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512