#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
##mpirun ./main ../datasets/DAMP/ECG/ECG1M.txt 1000000 64 128 ../results/discords/ECG1M_64_128_1cpu_K40.csv ../results/times/ECG1M_64_128_1cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG2M.txt 2000000 64 128 ../results/discords/ECG2M_64_128_1cpu_K40.csv ../results/times/ECG2M_64_128_1cpu_K40.csv

##mpirun ./main ../datasets/DAMP/ECG/ECG4M.txt 4000000 64 128 ../results/discords/ECG4M_64_128_1cpu_K40.csv ../results/times/ECG4M_64_128_1cpu_K40.csv

##mpirun ./main ../datasets/DAMP/RandomWalk/RandomWalk500K.txt 500000 64 128 ../results/discords/RandomWalk500K_64_128_1cpu_P100.csv ../results/times/RandomWalk500K_64_128_1cpu_P100.csv

##mpirun ./main ../datasets/DAMP/RandomWalk/RandomWalk1M.txt 1000000 64 128 ../results/discords/RandomWalk1M_64_128_1cpu_P100.csv ../results/times/RandomWalk1M_64_128_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/ECG/ECG500K.txt 500000 64 128 ../results/discords/ECG500K_64_128_1cpu_P100.csv ../results/times/ECG500K_64_128_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/ECG/ECG500K.txt 500000 128 256 ../results/discords/ECG500K_128_256_1cpu_P100.csv ../results/times/ECG500K_128_256_1cpu_P100.csv