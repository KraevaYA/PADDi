#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out
 
mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk250K.txt 250000 64 128 ../results/discords/RandomWalk1_250K_64_128_1cpu_P100.csv ../results/times/RandomWalk1_250K_64_128_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk500K.txt 500000 64 128 ../results/discords/RandomWalk1_500K_64_128_1cpu_P100.csv ../results/times/RandomWalk1_500K_64_128_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk1M.txt 1000000 64 128 ../results/discords/RandomWalk1_1M_64_128_1cpu_P100.csv ../results/times/RandomWalk1_1M_64_128_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk2M.txt 2000000 64 128 ../results/discords/RandomWalk1_2M_64_128_1cpu_P100.csv ../results/times/RandomWalk1_2M_64_128_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk4M.txt 4000000 64 128 ../results/discords/RandomWalk1_4M_64_128_1cpu_P100.csv ../results/times/RandomWalk1_4M_64_128_1cpu_P100.csv


mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk250K.txt 250000 128 256 ../results/discords/RandomWalk1_250K_128_256_1cpu_P100.csv ../results/times/RandomWalk1_250K_128_256_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk500K.txt 500000 128 256 ../results/discords/RandomWalk1_500K_128_256_1cpu_P100.csv ../results/times/RandomWalk1_500K_128_256_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk1M.txt 1000000 128 256 ../results/discords/RandomWalk1_1M_128_256_1cpu_P100.csv ../results/times/RandomWalk1_1M_128_256_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk2M.txt 2000000 128 256 ../results/discords/RandomWalk1_2M_128_256_1cpu_P100.csv ../results/times/RandomWalk1_2M_128_256_1cpu_P100.csv

mpirun ./main ../datasets/DAMP/RandomWalk1/RandomWalk4M.txt 4000000 128 256 ../results/discords/RandomWalk1_4M_128_256_1cpu_P100.csv ../results/times/RandomWalk1_4M_128_256_1cpu_P100.csv