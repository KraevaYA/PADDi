#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/power_demand/power_demand5000.txt 5000 128 200 ../results/discords/power_demand5000_128_200_2cpu_P100.csv ../results/times/power_demand5000_128_200_2cpu_P100.csv


##mpirun ./main ../datasets/power_demand/input0000.txt 5000 128 512