#!/bin/bash

#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
##SBATCH --time=12:00:00
#SBATCH --output=../output/slurm-%j.out


mpirun ./main ../datasets/VALMOD/GAP/GAP250K.txt 250000 256 512 ../results/discords/GAP250K_256_512_16cpu_K40.csv ../results/times/GAP250K_256_512_16cpu_K40.csv

mpirun ./main ../datasets/VALMOD/GAP/GAP500K.txt 500000 256 512 ../results/discords/GAP500K_256_512_16cpu_K40.csv ../results/times/GAP500K_256_512_16cpu_K40.csv

mpirun ./main ../datasets/VALMOD/GAP/GAP1M.txt 1000000 256 512 ../results/discords/GAP1M_256_512_16cpu_K40.csv ../results/times/GAP1M_256_512_16cpu_K40.csv

mpirun ./main ../datasets/VALMOD/GAP/GAP2M.txt 2000000 256 512 ../results/discords/GAP2M_256_512_16cpu_K40.csv ../results/times/GAP2M_256_512_16cpu_K40.csv


mpirun ./main ../datasets/DAMP/ECG/ECG250K.txt 250000 128 256 ../results/discords/ECG250K_128_256_16cpu_K40.csv ../results/times/ECG250K_128_256_16cpu_K40.csv

mpirun ./main ../datasets/DAMP/ECG/ECG500K.txt 500000 128 256 ../results/discords/ECG500K_128_256_16cpu_K40.csv ../results/times/ECG500K_128_256_16cpu_K40.csv
 
mpirun ./main ../datasets/DAMP/ECG/ECG1M.txt 1000000 128 256 ../results/discords/ECG1M_128_256_16cpu_K40.csv ../results/times/ECG1M_128_256_16cpu_K40.csv

mpirun ./main ../datasets/DAMP/ECG/ECG2M.txt 2000000 128 256 ../results/discords/ECG2M_128_256_16cpu_K40.csv ../results/times/ECG2M_128_256_16cpu_K40.csv

mpirun ./main ../datasets/DAMP/ECG/ECG4M.txt 4000000 128 256 ../results/discords/ECG4M_128_256_16cpu_K40.csv ../results/times/ECG4M_128_256_16cpu_K40.csv

mpirun ./main ../datasets/DAMP/ECG/ECG2M.txt 2000000 64 128 ../results/discords/ECG2M_64_128_16cpu_K40.csv ../results/times/ECG2M_64_128_16cpu_K40.csv

mpirun ./main ../datasets/DAMP/ECG/ECG4M.txt 4000000 64 128 ../results/discords/ECG4M_64_128_16cpu_K40.csv ../results/times/ECG4M_64_128_16cpu_K40.csv